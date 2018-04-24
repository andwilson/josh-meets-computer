import React from "react";
import Helmet from "react-helmet";
import styled from "styled-components";
import Img from "gatsby-image";

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 2fr;
  grid-column-gap: 10px;
`;

const Social = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  > h2 {
    margin: 5px 0 5px 0;
  }
  > p {
    color: grey;
    font-size: 12px;
    margin: 5px 0 5px 0;
  }
`;

const Description = styled.div`
  > p {
    @media (max-width: 600px) {
      font-size: 12px;
    }
  }
`;

const Title = styled.h1`
  color: black;
  margin: 0;
  > span {
    color: #28aa55;
  }
`;

const Avatar = styled(Img)`
  width: 150px;
  border-radius: 50%;
  margin-bottom: 0.5em;
  border: 1px solid grey;
  @media (max-width: 600px) {
    width: 110px;
  }
`;

export default ({ data }) => (
  <div>
    <Helmet>
      <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
      />
    </Helmet>
    <Grid>
      <Social>
        <Avatar sizes={data.avatar.sizes} />
        <h2>Contact</h2>
        <p>
          <i class="fa fa-cloud"></i>+1 (240) 418-4040
        </p>
        <p>j.a.zastrow.jr@gmail.com</p>
      </Social>
      <Description>
        <Title>
          Hi. My name is <span>Josh Zastrow</span>.
        </Title>
        <p>
          Need an enthusiastic engineer on your data science, machine learning
          or A.I team? Please reach out to me! I am always excited at the
          prospect of collaborating with other passionately driven people on
          bigger projects. Being a nomadic Engineer, I spend a good amount of
          time abroad, but am open for grabbing a coffee if you are in the San
          Francisco area.
        </p>
      </Description>
    </Grid>
  </div>
);

export const pageQuery = graphql`
  query AboutQuery {
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
