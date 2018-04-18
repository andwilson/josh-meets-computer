import React from "react";
import Link from "gatsby-link";
import get from "lodash/get";
import Helmet from "react-helmet";

import DefaultNav from "../components/DefaultNav";

class Notes extends React.Component {
  render() {
    const posts = get(this, "props.data.allMarkdownRemark.edges");

    return (
      <div>
        <Helmet title={get(this, "props.data.site.siteMetadata.title")} />
        {posts.map(post => {
          if (
            post.node.path !== "/404/" &&
            post.node.frontmatter.category == "Notes"
          ) {
            return (
              <div key={post.node.frontmatter.path}>
                <h3>
                  <Link to={post.node.frontmatter.path}>
                    {post.node.frontmatter.title}
                  </Link>
                </h3>
                <small>{post.node.frontmatter.date}</small>
                <p dangerouslySetInnerHTML={{ __html: post.node.excerpt }} />
              </div>
            );
          }
        })}
      </div>
    );
  }
}

Notes.propTypes = {
  route: React.PropTypes.object
};

export default Notes;

export const pageQuery = graphql`
  query NotesQuery {
    site {
      siteMetadata {
        title
      }
    }
    allMarkdownRemark(sort: { fields: [frontmatter___date], order: DESC }) {
      edges {
        node {
          excerpt
          frontmatter {
            path
            date(formatString: "DD MMMM, YYYY")
            title
            category
          }
        }
      }
    }
  }
`;
