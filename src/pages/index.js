import React from 'react'
import Link from 'gatsby-link'
import Helmet from 'react-helmet'
import Img from 'gatsby-image'
import styled from 'styled-components'

import HomeNav from '../components/HomeNav'

const HeroWrapper = styled.div`
  position: relative;
  height: 100vh;
  z-index: 1;
`;

const HeroContainer = styled.div`
  display: flex;
  height: 100vh;
  justify-content: center;
  align-items: center;
  position: relative;
  max-width: 960px;
  z-index: 2;
`;

export default ({ data }) => (
  <HeroWrapper>
    <Helmet title={data.site.siteMetadata.title} />
    <HeroContainer>
      <HomeNav data={data}/>
    </HeroContainer>
    <Img
      style={{
        minHeight: "100%",
        minWidth: "1024px",
        width: "100%",
        height: "auto",
        position: "fixed",
        top: 0,
        right: 0,
        opacity: 0.2
      }}
      sizes={data.background.sizes}
    />
  </HeroWrapper>
);

export const pageQuery = graphql`
  query IndexQuery {
    site {
      siteMetadata {
        title
      }
    }
    background: imageSharp(id: {regex: "/hex-bg.jpg/"}) {
      sizes(maxWidth: 1240, grayscale: false) {
        ...GatsbyImageSharpSizes
      }
    }
    boy: imageSharp(id: {regex: "/boy.png/"}) {
      sizes(maxWidth: 400, grayscale: false) {
        ...GatsbyImageSharpSizes
      }
    }
    computer: imageSharp(id: {regex: "/computer.png/"}) {
      sizes(maxWidth: 400, grayscale: false) {
        ...GatsbyImageSharpSizes
      }
    }
  }
`;
